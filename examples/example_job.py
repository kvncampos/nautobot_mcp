from nautobot.apps.jobs import Job, ObjectVar, StringVar, register_jobs
from nautobot.dcim.models import Device, DeviceType, Location, Role, Status


class CreateDeviceJob(Job):
    class Meta:
        name = "Create Device"
        description = "Creates a new device in Nautobot with user-specified parameters."
        field_order = ["device_name", "device_type", "location", "role"]

    device_name = StringVar(description="Name of the new device")
    device_type = ObjectVar(model=DeviceType, description="Device type")
    location = ObjectVar(model=Location, description="Device location")
    role = ObjectVar(model=Role, description="Device role")

    def run(self, *, device_name, device_type, location, role):
        # Permission check
        if not self.user.has_perm("dcim.add_device"):
            self.logger.failure("User does not have permission to create devices.")
            self.fail("Missing permission: dcim.add_device")
            return

        # Get status
        status_active = Status.objects.get(name="Active")

        # Create device
        device = Device(
            name=device_name,
            device_type=device_type,
            location=location,
            status=status_active,
            role=role,
        )
        device.validated_save()
        self.logger.success("Created device", extra={"object": device})

        return f"Device '{device.name}' created in location '{location.name}' with type '{device_type.model}' and role '{role.name}'."


register_jobs(CreateDeviceJob)
